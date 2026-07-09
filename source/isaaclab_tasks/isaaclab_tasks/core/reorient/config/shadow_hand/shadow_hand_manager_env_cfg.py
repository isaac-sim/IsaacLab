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
from isaaclab.sensors import JointWrenchSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.materials import RigidBodyMaterialBaseCfg
from isaaclab.utils.configclass import configclass

import isaaclab_tasks.core.reorient.mdp as mdp
from isaaclab_tasks.core.reorient.config.shadow_hand.shadow_hand_env_cfg import (
    NewtonEventCfg,
    ObjectCfg,
    PhysicsCfg,
    PhysxEventCfg,
    ShadowHandEnvCfg,
    ShadowHandOpenAIEnvCfg,
)
from isaaclab_tasks.core.reorient.reorient_task_constants import GOAL_MARKER_POSITION
from isaaclab_tasks.utils import PresetCfg

_DIRECT_CFG = ShadowHandEnvCfg()
_ACTUATED_JOINT_NAMES = _DIRECT_CFG.actuated_joint_names
_FINGERTIP_BODY_NAMES = _DIRECT_CFG.fingertip_body_names
_OPENAI_DIRECT_CFG = ShadowHandOpenAIEnvCfg()


@configclass
class _ShadowHandManagerSceneCfg(InteractiveSceneCfg):
    """Scene shared by the Shadow Hand Manager backend alternatives."""

    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    robot: PresetCfg = _DIRECT_CFG.robot_cfg
    object: ObjectCfg = _DIRECT_CFG.object_cfg
    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75)),
    )


@configclass
class ShadowHandManagerSceneCfg(PresetCfg):
    """Backend-specific scene cloning settings matching the Direct task."""

    physx = _ShadowHandManagerSceneCfg(num_envs=8192, env_spacing=0.75, replicate_physics=True, clone_in_fabric=True)
    newton_mjwarp = _ShadowHandManagerSceneCfg(
        num_envs=8192, env_spacing=0.75, replicate_physics=True, clone_in_fabric=False
    )
    ovphysx = _ShadowHandManagerSceneCfg(num_envs=8192, env_spacing=0.75, replicate_physics=True, clone_in_fabric=True)
    newton_kamino = newton_mjwarp
    default = physx


@configclass
class CommandsCfg:
    """Object pose goal matching the Direct in-hand target."""

    object_pose = mdp.ReorientEpisodeCommandCfg(
        asset_name="object",
        init_pos_offset=(0.0, 0.0, -0.04),
        update_goal_on_success=True,
        orientation_success_threshold=_DIRECT_CFG.success_tolerance,
        make_quat_unique=False,
        fixed_marker_pos=GOAL_MARKER_POSITION,
        goal_pose_visualizer_cfg=_DIRECT_CFG.goal_object_cfg,
        debug_vis=True,
    )


@configclass
class ActionsCfg:
    """Twenty actuated Shadow Hand joints."""

    joint_pos = mdp.EMAJointPositionToLimitsActionCfg(
        asset_name="robot",
        joint_names=_ACTUATED_JOINT_NAMES,
        alpha=_DIRECT_CFG.act_moving_average,
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
        scale=_DIRECT_CFG.vel_obs_scale,
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
        scale=_DIRECT_CFG.vel_obs_scale,
        params={"asset_cfg": SceneEntityCfg("object")},
    )
    goal_pose = ObsTerm(func=mdp.generated_commands, params={"command_name": "object_pose"})
    goal_quat_diff = ObsTerm(
        func=mdp.goal_quat_diff,
        params={"asset_cfg": SceneEntityCfg("object"), "command_name": "object_pose", "make_quat_unique": False},
    )
    fingertip_pos = ObsTerm(
        func=mdp.fingertip_pos,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=_FINGERTIP_BODY_NAMES, preserve_order=False)},
    )
    fingertip_quat = ObsTerm(
        func=mdp.fingertip_quat,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=_FINGERTIP_BODY_NAMES, preserve_order=False)},
    )
    fingertip_vel = ObsTerm(
        func=mdp.fingertip_vel,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=_FINGERTIP_BODY_NAMES, preserve_order=False)},
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
            "position_noise": _DIRECT_CFG.reset_position_noise,
            "joint_position_noise": _DIRECT_CFG.reset_dof_pos_noise,
            "joint_velocity_noise": _DIRECT_CFG.reset_dof_vel_noise,
            "action_name": "joint_pos",
        },
    )


@configclass
class RewardsCfg:
    """Direct-compatible reward and success accounting."""

    reorient = RewTerm(
        func=mdp.DirectReorientReward,
        weight=1.0,
        params={
            "command_name": "object_pose",
            "distance_scale": _DIRECT_CFG.dist_reward_scale,
            "rotation_scale": _DIRECT_CFG.rot_reward_scale,
            "rotation_epsilon": _DIRECT_CFG.rot_eps,
            "action_penalty_scale": _DIRECT_CFG.action_penalty_scale,
            "success_tolerance": _DIRECT_CFG.success_tolerance,
            "success_bonus": _DIRECT_CFG.reach_goal_bonus,
            "fall_distance": _DIRECT_CFG.fall_dist,
            "fall_penalty": _DIRECT_CFG.fall_penalty,
            "averaging_factor": _DIRECT_CFG.av_factor,
            "success_count_threshold": _DIRECT_CFG.success_count_threshold,
            "object_cfg": SceneEntityCfg("object"),
        },
    )


@configclass
class TerminationsCfg:
    """Termination conditions matching the Direct task."""

    object_out_of_reach = DoneTerm(
        func=mdp.object_reorientation_out_of_reach,
        params={
            "threshold": _DIRECT_CFG.fall_dist,
            "command_name": "object_pose",
            "object_cfg": SceneEntityCfg("object"),
        },
    )
    time_out = DoneTerm(func=mdp.direct_timeout, time_out=True)


@configclass
class ShadowHandManagerEnvCfg(ManagerBasedRLEnvCfg):
    """Manager-based state Shadow Hand task with Direct-compatible semantics."""

    scene: ShadowHandManagerSceneCfg = ShadowHandManagerSceneCfg()
    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 120.0,
        render_interval=2,
        physics_material=RigidBodyMaterialBaseCfg(static_friction=1.0, dynamic_friction=1.0),
        physics=PhysicsCfg(),
    )
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        self.decimation = _DIRECT_CFG.decimation
        self.episode_length_s = _DIRECT_CFG.episode_length_s
        self.sim.render_interval = self.decimation
        self.viewer.eye = (2.0, 2.0, 2.0)


@configclass
class OpenAICommandsCfg:
    """OpenAI goal command with its wider success tolerance."""

    object_pose = mdp.ReorientEpisodeCommandCfg(
        asset_name="object",
        init_pos_offset=(0.0, 0.0, -0.04),
        update_goal_on_success=True,
        orientation_success_threshold=_OPENAI_DIRECT_CFG.success_tolerance,
        make_quat_unique=False,
        fixed_marker_pos=GOAL_MARKER_POSITION,
        goal_pose_visualizer_cfg=_OPENAI_DIRECT_CFG.goal_object_cfg,
        debug_vis=True,
    )


@configclass
class OpenAIActionsCfg:
    """OpenAI actions with Direct-compatible EMA and stateful noise."""

    joint_pos = mdp.NoisyEMAJointPositionToLimitsActionCfg(
        asset_name="robot",
        joint_names=_ACTUATED_JOINT_NAMES,
        alpha=_OPENAI_DIRECT_CFG.act_moving_average,
        rescale_to_limits=True,
        noise_model=_OPENAI_DIRECT_CFG.action_noise_model,
    )


@configclass
class OpenAIObservationsCfg:
    """OpenAI 42-dimensional actor and 187-dimensional critic observations."""

    @configclass
    class PolicyCfg(ObsGroup):
        openai = ObsTerm(
            func=mdp.OpenAIPolicyObservation,
            params={
                "command_name": "object_pose",
                "action_name": "joint_pos",
                "noise_model": _OPENAI_DIRECT_CFG.observation_noise_model,
                "robot_cfg": SceneEntityCfg("robot", body_names=_FINGERTIP_BODY_NAMES, preserve_order=False),
                "object_cfg": SceneEntityCfg("object"),
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class CriticCfg(FullStateWithoutActionCfg):
        fingertip_wrench = ObsTerm(
            func=mdp.fingertip_wrench,
            scale=_OPENAI_DIRECT_CFG.force_torque_obs_scale,
            params={
                "sensor_cfg": SceneEntityCfg("joint_wrench", body_names=_FINGERTIP_BODY_NAMES, preserve_order=False)
            },
        )
        last_action = ObsTerm(func=mdp.reorient_last_action, params={"action_name": "joint_pos"})

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class _ShadowHandOpenAIManagerSceneCfg(_ShadowHandManagerSceneCfg):
    """Shadow Hand scene with fingertip joint-wrench sensing."""

    joint_wrench = JointWrenchSensorCfg(prim_path="{ENV_REGEX_NS}/Robot")


@configclass
class ShadowHandOpenAIManagerSceneCfg(PresetCfg):
    """Backend-specific OpenAI scene alternatives."""

    physx = _ShadowHandOpenAIManagerSceneCfg(
        num_envs=8192, env_spacing=0.75, replicate_physics=True, clone_in_fabric=True
    )
    newton_mjwarp = _ShadowHandOpenAIManagerSceneCfg(
        num_envs=8192, env_spacing=0.75, replicate_physics=True, clone_in_fabric=False
    )
    ovphysx = _ShadowHandOpenAIManagerSceneCfg(
        num_envs=8192, env_spacing=0.75, replicate_physics=True, clone_in_fabric=True
    )
    newton_kamino = newton_mjwarp
    default = physx


_OPENAI_RESET_PARAMS = {
    "position_noise": _OPENAI_DIRECT_CFG.reset_position_noise,
    "joint_position_noise": _OPENAI_DIRECT_CFG.reset_dof_pos_noise,
    "joint_velocity_noise": _OPENAI_DIRECT_CFG.reset_dof_vel_noise,
    "action_name": "joint_pos",
}


@configclass
class OpenAIPhysxEventCfg(PhysxEventCfg):
    """PhysX OpenAI randomization and state reset events."""

    reset_state = EventTerm(func=mdp.reset_reorient_state, mode="reset", params=_OPENAI_RESET_PARAMS)


@configclass
class OpenAINewtonEventCfg(NewtonEventCfg):
    """Newton OpenAI randomization and state reset events."""

    reset_state = EventTerm(func=mdp.reset_reorient_state, mode="reset", params=_OPENAI_RESET_PARAMS)


@configclass
class OpenAIEventCfg(PresetCfg):
    """Backend-specific OpenAI event alternatives."""

    physx = OpenAIPhysxEventCfg()
    newton_mjwarp = OpenAINewtonEventCfg()
    ovphysx = OpenAIPhysxEventCfg()
    newton_kamino = newton_mjwarp
    default = physx


@configclass
class OpenAIRewardsCfg:
    """Direct-compatible OpenAI reward and success accounting."""

    reorient = RewTerm(
        func=mdp.DirectReorientReward,
        weight=1.0,
        params={
            "command_name": "object_pose",
            "distance_scale": _OPENAI_DIRECT_CFG.dist_reward_scale,
            "rotation_scale": _OPENAI_DIRECT_CFG.rot_reward_scale,
            "rotation_epsilon": _OPENAI_DIRECT_CFG.rot_eps,
            "action_penalty_scale": _OPENAI_DIRECT_CFG.action_penalty_scale,
            "success_tolerance": _OPENAI_DIRECT_CFG.success_tolerance,
            "success_bonus": _OPENAI_DIRECT_CFG.reach_goal_bonus,
            "fall_distance": _OPENAI_DIRECT_CFG.fall_dist,
            "fall_penalty": _OPENAI_DIRECT_CFG.fall_penalty,
            "averaging_factor": _OPENAI_DIRECT_CFG.av_factor,
            "success_count_threshold": _OPENAI_DIRECT_CFG.success_count_threshold,
            "action_name": "joint_pos",
            "object_cfg": SceneEntityCfg("object"),
        },
    )


@configclass
class OpenAITerminationsCfg:
    """Direct-compatible OpenAI termination conditions."""

    object_out_of_reach = DoneTerm(
        func=mdp.object_reorientation_out_of_reach,
        params={
            "threshold": _OPENAI_DIRECT_CFG.fall_dist,
            "command_name": "object_pose",
            "object_cfg": SceneEntityCfg("object"),
        },
    )
    time_out = DoneTerm(
        func=mdp.direct_reorient_timeout,
        time_out=True,
        params={
            "command_name": "object_pose",
            "reward_name": "reorient",
            "success_tolerance": _OPENAI_DIRECT_CFG.success_tolerance,
            "max_successes": _OPENAI_DIRECT_CFG.max_consecutive_success,
            "object_cfg": SceneEntityCfg("object"),
        },
    )


@configclass
class ShadowHandOpenAIManagerEnvCfg(ShadowHandManagerEnvCfg):
    """Manager counterpart shared by the OpenAI FF and LSTM variants."""

    scene: ShadowHandOpenAIManagerSceneCfg = ShadowHandOpenAIManagerSceneCfg()
    observations: OpenAIObservationsCfg = OpenAIObservationsCfg()
    actions: OpenAIActionsCfg = OpenAIActionsCfg()
    commands: OpenAICommandsCfg = OpenAICommandsCfg()
    rewards: OpenAIRewardsCfg = OpenAIRewardsCfg()
    terminations: OpenAITerminationsCfg = OpenAITerminationsCfg()
    events: OpenAIEventCfg = OpenAIEventCfg()

    def __post_init__(self):
        super().__post_init__()
        self.decimation = _OPENAI_DIRECT_CFG.decimation
        self.episode_length_s = _OPENAI_DIRECT_CFG.episode_length_s
        self.sim.dt = _OPENAI_DIRECT_CFG.sim.dt
        self.sim.render_interval = self.decimation
        self.sim.physics = PhysicsCfg()
